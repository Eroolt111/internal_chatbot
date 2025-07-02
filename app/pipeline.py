import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional
import logging
from datetime import datetime

from llama_index.core import SQLDatabase, VectorStoreIndex, load_index_from_storage, Document
from llama_index.core.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from llama_index.core.retrievers import SQLRetriever
from llama_index.core.query_pipeline import QueryPipeline as QP, InputComponent, FnComponent
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.schema import TextNode
from llama_index.core.storage import StorageContext
from llama_index.core.llms import ChatResponse
from sqlalchemy import text
    
from .config import config
from .db import db_manager
from .llm import llm_manager
from .prompts import prompt_manager

logger = logging.getLogger(__name__)

class TableInfo(BaseModel):
    """Information regarding a structured table."""
    table_name: str = Field(..., description="table name (must be underscores and NO spaces)")
    table_summary: str = Field(..., description="short, concise summary/caption of the table")
    column_descriptions: dict = Field(..., description="dictionary of column_name: description")

class IndexTracker:
    """Track which database rows have been indexed"""
    def __init__(self, tracker_file: str = None):
        if tracker_file is None:
            tracker_file = Path(config.TABLE_INDEX_DIR) / "index_tracker.json"
        self.tracker_file = Path(tracker_file)
        self.tracker_file.parent.mkdir(parents=True, exist_ok=True)
        self.load_tracker()

    def load_tracker(self):
        try:
            with open(self.tracker_file, 'r', encoding='utf-8') as f:
                self.tracked = json.load(f)
        except FileNotFoundError:
            self.tracked = {}

    def save_tracker(self):
        with open(self.tracker_file, 'w', encoding='utf-8') as f:
            json.dump(self.tracked, f, indent=2, ensure_ascii=False)

    def get_last_indexed_id(self, table_name: str) -> int:
        return self.tracked.get(table_name, {}).get('last_id', 0)

    def get_last_indexed_count(self, table_name: str) -> int:
        return self.tracked.get(table_name, {}).get('last_count', 0)

    def update_last_indexed(self, table_name: str, last_id: int = None, last_count: int = None):
        if table_name not in self.tracked:
            self.tracked[table_name] = {}
        if last_id is not None:
            self.tracked[table_name]['last_id'] = last_id
        if last_count is not None:
            self.tracked[table_name]['last_count'] = last_count
        self.tracked[table_name]['last_update'] = datetime.now().isoformat()
        self.save_tracker()

class ChatbotPipeline:
    """Main chatbot pipeline for text-to-SQL and response generation"""
    def __init__(self):
        self.sql_database = None
        self.query_pipeline = None
        self.table_infos = []
        self.vector_index_dict = {}
        self.index_tracker = IndexTracker()
        self._initialize()

    def _initialize(self):
        logger.info("Initializing chatbot pipeline...")
        self.sql_database = SQLDatabase(db_manager.engine)
        self._generate_table_summaries()
        self._create_vector_indices()
        self._setup_query_pipeline() 
        logger.info("✅ Chatbot pipeline initialized successfully")

    def is_first_run(self) -> bool:
        return len(self.index_tracker.tracked) == 0

    def incremental_update(self) -> int:
        total_new_docs = 0
        for table_name in self.sql_database.get_usable_table_names():
            total_new_docs += self._update_table_index(table_name)
        logger.info(f"✅ Incremental update complete. Added {total_new_docs} new documents")
        return total_new_docs

    def _update_table_index(self, table_name: str) -> int:
        try:
            logger.debug(f"[DEBUG] Entering _update for {table_name}")
            with db_manager.get_connection() as conn:
                current_count = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar()
                last_count = self.index_tracker.get_last_indexed_count(table_name)
                last_id    = self.index_tracker.get_last_indexed_id(table_name)
                logger.debug(f"[DEBUG] {table_name}: current_count={current_count}, last_count={last_count}, last_id={last_id}")
                cols = conn.execute(text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = :tbl AND column_name IN ('id','ID','Id')
                    ORDER BY ordinal_position
                    """), {"tbl": table_name}).fetchall()
            id_col = cols[0][0] if cols else None
            last_id = self.index_tracker.get_last_indexed_id(table_name)
            last_count = self.index_tracker.get_last_indexed_count(table_name)
            if current_count <= last_count:
                return 0
            idx_path = Path(config.TABLE_INDEX_DIR) / table_name
            if not idx_path.exists():
                return self._create_full_table_index(table_name)
            idx = load_index_from_storage(StorageContext.from_defaults(persist_dir=str(idx_path)), index_id="vector_index")
            if id_col:
                new_rows = db_manager.get_new_rows_since_id(table_name, last_id, id_column=id_col, limit=config.MAX_ROWS_PER_TABLE)
            else:
                new_rows = db_manager.get_new_rows_by_offset(table_name, offset=last_count, limit=config.MAX_ROWS_PER_TABLE)
            
            logger.debug(f"[DEBUG] {table_name}: fetched {len(new_rows)} new rows -> {new_rows}")
            if not new_rows:
                return 0
            for row in new_rows:
                idx.insert(Document(text=str(row)))
            idx.storage_context.persist(str(idx_path))
            self.vector_index_dict[table_name] = idx
            self.index_tracker.update_last_indexed(table_name, last_count=current_count)
            if id_col:
                max_id = max(row[id_col] for row in new_rows)
                self.index_tracker.update_last_indexed(table_name, last_id=max_id)
            return len(new_rows)
        except Exception as e:
            logger.error(f"Error updating index for {table_name}: {e}")
            return 0

    def _create_full_table_index(self, table_name: str) -> int:
        """Create a full index for a table (used when index doesn't exist)"""
        try:
            idx_path = Path(config.TABLE_INDEX_DIR) / table_name
            with db_manager.get_connection() as conn:
                # Fetch all rows, not just a subset
                result = conn.execute(text(f'SELECT * FROM "{table_name}"'))
                rows = result.mappings().all()
                total_count = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar()
                # Detect ID column
                cols = conn.execute(text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = :tbl AND column_name IN ('id','ID','Id')
                    ORDER BY ordinal_position
                    """
                ), {"tbl": table_name}).fetchall()
            id_col = cols[0][0] if cols else None
            # Build vector nodes for all rows
            nodes = [Document(text=str(r)) for r in rows]
            idx = VectorStoreIndex(nodes)
            idx.set_index_id("vector_index")
            idx.storage_context.persist(str(idx_path))

            # Update tracker to reflect entire table
            last_id = max(r[id_col] for r in rows) if id_col and rows else None
            self.index_tracker.update_last_indexed(
                table_name,
                last_id=last_id,
                last_count=total_count
            )
            self.vector_index_dict[table_name] = idx
            logger.info(f"✅ Created full index for {table_name} with {len(nodes)} documents")
            return len(nodes)
        except Exception as e:
            logger.error(f"Error creating full index for {table_name}: {e}")
            return 0

    
    def _generate_table_summaries(self):
        logger.info("Generating table summaries...")
        dfs, table_info_list = db_manager.load_all_tables(limit=1000)
        program = LLMTextCompletionProgram.from_defaults(
            output_cls=TableInfo,
            llm=llm_manager.get_llm(),
            prompt_template_str=prompt_manager.get_table_info_prompt().template,
        )
        table_names = set()
        for idx, (df, table_info) in enumerate(zip(dfs, table_info_list)):
            existing = self._get_existing_table_info(idx)
            if existing:
                self.table_infos.append({
                    'original_table_name': table_info['table_name'],
                    'table_name': existing.table_name,
                    'table_summary': existing.table_summary,
                    'column_descriptions': existing.column_descriptions
                })
                logger.info(f"Loaded existing info for table: {existing.table_name}")
            else:
                table_structure = ", ".join(table_info['columns'])
                sample_data = df.head(5).to_string()
                attempts = 0
                while attempts < 3:
                    try:
                        gen = program(
                            table_name=table_info['table_name'],
                            table_structure=table_structure,
                            table_data=sample_data,
                            exclude_table_name_list=str(list(table_names)),
                        )
                        if gen.table_name not in table_names:
                            table_names.add(gen.table_name)
                            break
                        attempts += 1
                        logger.warning(f"Duplicate table_name {gen.table_name}, retrying...")
                    except Exception as e:
                        logger.error(f"Error generating table summary: {e}")
                        gen = TableInfo(
                            table_name=table_info['table_name'],
                            table_summary=f"Database table with {table_info['row_count']} rows",
                            column_descriptions={}
                        )
                        break
                self._save_table_info(idx, gen)
                self.table_infos.append({
                    'original_table_name': table_info['table_name'],
                    'table_name': gen.table_name,
                    'table_summary': gen.table_summary,
                    'column_descriptions': gen.column_descriptions
                })
                logger.info(f"Generated summary for table: {gen.table_name}")

    def _get_existing_table_info(self, idx: int) -> Optional[TableInfo]:
        results = list(Path(config.TABLE_INFO_DIR).glob(f"{idx}_*"))
        if not results:
            return None
        if len(results) == 1:
            try:
                data = json.loads(results[0].read_text(encoding='utf-8'))
                data.setdefault("column_descriptions", {})
                return TableInfo.model_validate(data)
            except Exception as e:
                logger.error(f"Error loading table info from {results[0]}: {e}")
        else:
            logger.warning(f"Multiple table info files for index {idx}: {results}")
        return None

    def _save_table_info(self, idx: int, info: TableInfo):
        out = Path(config.TABLE_INFO_DIR) / f"{idx}_{info.table_name}.json"
        try:
            out.write_text(json.dumps(info.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            logger.error(f"Error saving table info to {out}: {e}")

    def _create_vector_indices(self):
        logger.info("Creating vector indices for tables...")
        for tbl in self.sql_database.get_usable_table_names():
            logger.info(f"Indexing rows in table: {tbl}")
            idx_path = Path(config.TABLE_INDEX_DIR) / tbl
            if not idx_path.exists():
                self._create_full_table_index(tbl)
            else:
                try:
                    ctx = StorageContext.from_defaults(persist_dir=str(idx_path))
                    idx = load_index_from_storage(ctx, index_id="vector_index")
                    self.vector_index_dict[tbl] = idx
                except Exception as e:
                    logger.error(f"Error loading index for {tbl}: {e}")
                    self._create_full_table_index(tbl)
        logger.info(f"Created vector indices for {len(self.vector_index_dict)} tables")

    # _debug_sql_results function added as a class method ---
    def _debug_sql_results(self, sql_results) -> str:
        print(f"\n--- DEBUG: RAW SQL RESULTS (FROM DATABASE) ---\n{sql_results}\n--- END DEBUG ---\n")
        return sql_results 

    def _log_sql_query(self, sql_query: str) -> str:
        """Log the raw SQL before execution."""
        # print for debug output ---
        print(f"\n--- DEBUG: GENERATED SQL QUERY ---\n{sql_query}\n--- END DEBUG ---\n")
        logger.debug(f"→ running SQL:\n{sql_query}")
        return sql_query

    def _parse_response_to_sql(self, response: ChatResponse) -> str:
        """
        Extract a clean SQL statement from the LLM's response, stripping out
        markdown fences, any "assistant:" prefix, markers like SQLQuery/SQLResult,
        and anything after the final semicolon (including trailing prose or "Answer:").
        """
        #  print for debug output ---
        print(f"\n--- DEBUG: LLM Response BEFORE SQL Parsing ---\n{response.message.content.strip()}\n--- END DEBUG ---\n")
        
        txt = response.message.content.strip()

        # Drop leading "assistant:" if present
        if txt.lower().startswith("assistant:"):
            txt = txt[len("assistant:"):].strip()

        # If there's a fenced sql block, grab its contents
        fence = re.search(r"```(?:sql)?\s*([\s\S]*?)```", txt, re.IGNORECASE)
        if fence:
            txt = fence.group(1).strip()
        else:
            # Otherwise just remove stray backticks
            txt = txt.replace("```", "").replace("`", "")

        # Remove any SQLQuery: or SQLResult: sections
        txt = re.sub(r"(?i)sqlquery:.*", "", txt)
        txt = re.sub(r"(?i)sqlresult:.*", "", txt)

        # Truncate at the last semicolon (keep the semicolon)
        if ";" in txt:
            txt = txt[: txt.rfind(";") + 1]

        # Remove any trailing "Answer:" or similar explanation
        txt = re.split(r"(?i)\banswer\s*:", txt)[0].strip()

        # Finally drop any leading dialect label (sql:, postgresql:)
        txt = re.sub(r"^(?:sql|postgresql)[:\s]*", "", txt, flags=re.IGNORECASE).strip()

        # print for debug output ---
        print(f"\n--- DEBUG: PARSED SQL (After Cleaning) ---\n{txt}\n--- END DEBUG ---\n")
        return txt

    def _setup_query_pipeline(self):
        logger.info("Setting up query pipeline...")
        # Build schema contexts
        schemas = []
        for t in self.table_infos:
            cols = "\n".join(f"- {c}: {d}" for c, d in t['column_descriptions'].items())
            ctx = f"Descriptive name: {t['table_name']}. {t['table_summary']}\nColumns:\n{cols}"
            schemas.append(SQLTableSchema(table_name=t['original_table_name'], context_str=ctx))

        node_map = SQLTableNodeMapping(self.sql_database)
        obj_index = ObjectIndex.from_objects(schemas, node_map, VectorStoreIndex)
        table_retriever = obj_index.as_retriever(similarity_top_k=config.MAX_TABLE_RETRIEVAL)

        sql_retriever = SQLRetriever(self.sql_database)
        table_parser = FnComponent(fn=self._get_table_context_and_rows_str)
        sql_parser  = FnComponent(fn=self._parse_response_to_sql)
        log_sql     = FnComponent(fn=self._log_sql_query)

        self.query_pipeline = QP(verbose=config.DEBUG) 
        self.query_pipeline.add_modules({
            "input": InputComponent(),
            "table_retriever": table_retriever,
            "table_output_parser": table_parser,
            "text2sql_prompt": prompt_manager.get_text2sql_prompt(),
            "text2sql_llm": llm_manager.get_llm(),
            "sql_output_parser": sql_parser,
            "log_sql": log_sql,
            "sql_retriever": sql_retriever,
            # debug_sql_results_printer module
            "debug_sql_results_printer": FnComponent(fn=self._debug_sql_results), 
            "response_synthesis_prompt": prompt_manager.get_response_synthesis_prompt(),
            "response_synthesis_llm": llm_manager.get_llm(),
        })

        self.query_pipeline.add_link("input", "table_retriever")
        self.query_pipeline.add_link("input", "table_output_parser", dest_key="query_str")
        self.query_pipeline.add_link("table_retriever", "table_output_parser", dest_key="table_schema_objs")
        self.query_pipeline.add_link("input", "text2sql_prompt", dest_key="query_str")
        self.query_pipeline.add_link("table_output_parser", "text2sql_prompt", dest_key="schema")
        self.query_pipeline.add_chain([
            "text2sql_prompt",
            "text2sql_llm",
            "sql_output_parser",
            "log_sql",
            "sql_retriever",
            # Added debug_sql_results_printer to the chain ---
            "debug_sql_results_printer", 
        ])
        self.query_pipeline.add_link("sql_output_parser", "response_synthesis_prompt", dest_key="sql_query")
        
        # Changed link to use debug_sql_results_printer ---
        self.query_pipeline.add_link("debug_sql_results_printer", "response_synthesis_prompt", dest_key="context_str")
        
        self.query_pipeline.add_link("input", "response_synthesis_prompt", dest_key="query_str")
        self.query_pipeline.add_link("response_synthesis_prompt", "response_synthesis_llm")

    def _get_table_context_and_rows_str(self, query_str: str, table_schema_objs: List[SQLTableSchema]) -> str:
        parts = []
        for schema in table_schema_objs:
            # Get basic table info
            info = self.sql_database.get_single_table_info(schema.table_name)
        
            # Add detailed table description with column information
            if schema.context_str:
                info += f"\n\nTable Description: {schema.context_str}"
        
            # Find the matching table info for detailed column descriptions
            table_info = None
            for t in self.table_infos:
                if t['original_table_name'] == schema.table_name:
                    table_info = t
                    break
        
            # Add detailed column descriptions
            if table_info and table_info['column_descriptions']:
                info += "\n\nDetailed Column Descriptions:"
                for col_name, col_desc in table_info['column_descriptions'].items():
                    info += f"\n- {col_name}: {col_desc}"
        
            # Add sample rows with context about multi-column relationships
            if schema.table_name in self.vector_index_dict:
                try:
                    retr = self.vector_index_dict[schema.table_name].as_retriever(
                        similarity_top_k=config.MAX_ROW_RETRIEVAL
                    )
                    nodes = retr.retrieve(query_str)
                    if nodes:
                        info += "\n\nRelevant Example Rows (Note: Many records require multiple column conditions):"
                        for i, node in enumerate(nodes):
                            content = str(node.get_content())
                            info += f"\nExample {i+1}: {content}"
                    
                        # Add guidance about multi-column filtering
                        info += "\n\nIMPORTANT: When filtering this data, consider that records are often distinguished by combinations of columns like (period + scr_mn + scr_eng) or (period + code + scr_mn). A single column filter may not be sufficient to get the exact record you need."
                        
                except Exception as e:
                    logger.error(f"Error retrieving rows for {schema.table_name}: {e}")
        
            parts.append(info)
    
        return "\n\n" + "="*50 + "\n\n".join(parts)

    def _parse_response_to_sql(self, response: ChatResponse) -> str:
        """
        Extract a clean SQL statement from the LLM's response, stripping out
        markdown fences, any "assistant:" prefix, markers like SQLQuery/SQLResult,
        and anything after the final semicolon (including trailing prose or "Answer:").
        """
        #  print for debug output ---
        print(f"\n--- DEBUG: LLM Response BEFORE SQL Parsing ---\n{response.message.content.strip()}\n--- END DEBUG ---\n")

        txt = response.message.content.strip()

        #  Drop leading "assistant:" if present
        if txt.lower().startswith("assistant:"):
            txt = txt[len("assistant:"):].strip()

        #  If there's a fenced sql block, grab its contents
        fence = re.search(r"```(?:sql)?\s*([\s\S]*?)```", txt, re.IGNORECASE)
        if fence:
            txt = fence.group(1).strip()
        else:
            # Otherwise just remove stray backticks
            txt = txt.replace("```", "").replace("`", "")

        #Remove any SQLQuery: or SQLResult: sections
        txt = re.sub(r"(?i)sqlquery:.*", "", txt)
        txt = re.sub(r"(?i)sqlresult:.*", "", txt)

        #Truncate at the last semicolon (keep the semicolon)
        if ";" in txt:
            txt = txt[: txt.rfind(";") + 1]

        # Remove any trailing "Answer:" or similar explanation
        txt = re.split(r"(?i)\banswer\s*:", txt)[0].strip()

        #drop any leading dialect label (sql:, postgresql:)
        txt = re.sub(r"^(?:sql|postgresql)[:\s]*", "", txt, flags=re.IGNORECASE).strip()

        print(f"\n--- DEBUG: PARSED SQL (After Cleaning) ---\n{txt}\n--- END DEBUG ---\n")
        return txt

    def refresh_indices(self) -> Dict[str, int]:
        """
        Refresh in-memory indices from disk storage.
        Returns dict of table_name -> number of documents loaded.
        """
        logger.info("Refreshing indices from disk...")
        refreshed_counts = {}
    
        # Reload the tracker from disk
        self.index_tracker.load_tracker()
    
        # Reload all vector indices from disk
        for table_name in self.sql_database.get_usable_table_names():
            idx_path = Path(config.TABLE_INDEX_DIR) / table_name
            if idx_path.exists():
                try:
                    # Load the updated index from disk
                    ctx = StorageContext.from_defaults(persist_dir=str(idx_path))
                    idx = load_index_from_storage(ctx, index_id="vector_index")
                    self.vector_index_dict[table_name] = idx
                
                    # Count documents (approximate)
                    # Note: LlamaIndex doesn't have a direct doc count method
                    # This is an approximation
                    try:
                        # Try to get some measure of index size
                        retriever = idx.as_retriever(similarity_top_k=1)
                        test_nodes = retriever.retrieve("test")
                        refreshed_counts[table_name] = "refreshed"
                    except:
                        refreshed_counts[table_name] = "refreshed"
                    
                    logger.info(f"Refreshed index for table: {table_name}")
                except Exception as e:
                    logger.error(f"Error refreshing index for {table_name}: {e}")
                    refreshed_counts[table_name] = f"error: {e}"
            else:
                logger.warning(f"No index found for table {table_name}")
    
        logger.info(f"Index refresh complete. Refreshed {len(refreshed_counts)} tables")
        return refreshed_counts

def get_index_status(self) -> Dict[str, Dict]:
    """
    Get status information about all indices.
    Returns dict with table names and their indexing status.
    """
    status = {}
    
    for table_name in self.sql_database.get_usable_table_names():
        try:
            with db_manager.get_connection() as conn:
                current_count = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar()
            
            last_count = self.index_tracker.get_last_indexed_count(table_name)
            last_id = self.index_tracker.get_last_indexed_id(table_name)
            
            idx_path = Path(config.TABLE_INDEX_DIR) / table_name
            index_exists = idx_path.exists()
            
            status[table_name] = {
                'current_db_count': current_count,
                'last_indexed_count': last_count,
                'last_indexed_id': last_id,
                'index_exists': index_exists,
                'needs_update': current_count > last_count,
                'pending_rows': max(0, current_count - last_count)
            }
        except Exception as e:
            status[table_name] = {'error': str(e)}
    
    return status

def auto_refresh_if_needed(self) -> bool:
    """
    Check if indices need refreshing and refresh them if needed.
    Returns True if refresh was performed.
    """
    # Check if tracker file has been modified since we last loaded it
    tracker_file = Path(self.index_tracker.tracker_file)
    if not tracker_file.exists():
        return False
    
    # Get file modification time
    file_mtime = tracker_file.stat().st_mtime
    
    # Check if we have a stored mtime and if file is newer
    if not hasattr(self, '_last_tracker_mtime'):
        self._last_tracker_mtime = file_mtime
        return False
    
    if file_mtime > self._last_tracker_mtime:
        logger.info("Tracker file updated externally, refreshing indices...")
        self.refresh_indices()
        self._last_tracker_mtime = file_mtime
        return True
    
    return False

def run_query(self, query_str: str):
    """Run a query through the complete pipeline with auto-refresh"""
    try:
        # Check if indices need refreshing before running query
        self.auto_refresh_if_needed()
        
        return self.query_pipeline.run(input=query_str) 

    except Exception as e:
        logger.error(f"Error running query: {e}")
        raise