import psycopg2
import pandas as pd
from sqlalchemy import create_engine, MetaData, inspect, text
from sqlalchemy.engine import Engine
from typing import List, Dict, Optional, Tuple
import logging
from contextlib import contextmanager
 
from .config import config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database connection and management class"""
    
    def __init__(self):
        self.engine: Optional[Engine] = None
        self._connect()
    
    def _connect(self) -> None:
        """Create SQLAlchemy engine for PostgreSQL"""
        try:
            self.engine = create_engine(
                config.DATABASE_URL,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version();"))
                version = result.fetchone()[0]
                logger.info(f"Connected to PostgreSQL: {version}")
                
        except Exception as e:
            logger.error(f"Error creating database engine: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        if not self.engine:
            raise RuntimeError("Database engine not initialized")
        
        connection = self.engine.connect()
        try:
            yield connection
        finally:
            connection.close()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_connection() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_table_names(self) -> List[str]:
        """Get list of table names in the database"""
        try:
            inspector = inspect(self.engine)
            return inspector.get_table_names()
        except Exception as e:
            logger.error(f"Error getting table names: {e}")
            return []
    
    def get_table_info(self, table_name: str) -> Dict:
        """Get detailed information about a table"""
        try:
            inspector = inspect(self.engine)
            columns = inspector.get_columns(table_name)
            
            # Get row count
            with self.get_connection() as conn:
                result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
                row_count = result.fetchone()[0]
            
            return {
                'table_name': table_name,
                'columns': [f"{col['name']} ({col['type']})" for col in columns],
                'column_names': [col['name'] for col in columns],
                'row_count': row_count
            }
        except Exception as e:
            logger.error(f"Error getting table info for {table_name}: {e}")
            return {}
    
    def load_table_data(self, table_name: str, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Load data from a table"""
        try:
            query = f'SELECT * FROM "{table_name}" LIMIT {limit}'
            df = pd.read_sql(query, self.engine)
            logger.info(f"Loaded {len(df)} rows from table '{table_name}'")
            return df
        except Exception as e:
            logger.error(f"Error loading data from table {table_name}: {e}")
            return None
    
    def load_all_tables(self, limit: int = 1000) -> Tuple[List[pd.DataFrame], List[Dict]]:
        """Load data from all tables"""
        table_names = self.get_table_names()
        dfs = []
        table_info_list = []
        
        for table_name in table_names:
            df = self.load_table_data(table_name, limit)
            if df is not None:
                dfs.append(df)
                table_info = self.get_table_info(table_name)
                table_info_list.append(table_info)
        
        return dfs, table_info_list
    
    def execute_query(self, query: str) -> List[Dict]:
        """Execute a SQL query and return results"""
        try:
            with self.get_connection() as conn:
                result = conn.execute(text(query))
                columns = result.keys()
                rows = result.fetchall()
                
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            logger.error(f"Query: {query}")
            raise
    
    def get_sample_data(self, table_name: str, num_rows: int = 5) -> str:
        """Get sample data from a table as string"""
        try:
            df = self.load_table_data(table_name, limit=num_rows)
            if df is not None:
                return df.to_string()
            return ""
        except Exception as e:
            logger.error(f"Error getting sample data from {table_name}: {e}")
            return ""

    # -------------------------------------------------------------------------
    # Incremental indexing support methods
    # -------------------------------------------------------------------------
    def get_new_rows_since_id(
        self,
        table_name: str,
        last_id: int,
        id_column: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Fetch new rows from `table_name` where id_column > last_id.
        Auto-detects id column if not provided (looks for 'id','ID','Id').
        Returns list of dicts.
        """
        with self.get_connection() as conn:
            # Auto-detect id column if needed
            if id_column is None:
                cols = conn.execute(text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = :table
                      AND column_name IN ('id','ID','Id')
                    ORDER BY ordinal_position
                    """
                ), {"table": table_name}).fetchall()
                if not cols:
                    raise ValueError(f"No ID column found for table {table_name}")
                id_column = cols[0][0]

            # Build and execute query
            sql = f'SELECT * FROM "{table_name}" WHERE "{id_column}" > :last_id ORDER BY "{id_column}"'
            if limit:
                sql += f' LIMIT {limit}'

            result = conn.execute(text(sql), {"last_id": last_id})
            # Convert rows to dicts via RowMapping
            rows = result.mappings().all()
            return [dict(r) for r in rows]

    def get_new_rows_by_offset(
        self,
        table_name: str,
        offset: int,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Fetch rows from `table_name` starting at `offset`.
        Returns list of dicts.
        """
        with self.get_connection() as conn:
            sql = f'SELECT * FROM "{table_name}" OFFSET :offset'
            if limit:
                sql += f' LIMIT {limit}'

            result = conn.execute(text(sql), {"offset": offset})
            # Use RowMapping to get dict-like rows
            rows = result.mappings().all()
            return [dict(r) for r in rows]


db_manager = DatabaseManager()
