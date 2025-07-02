// Chat Application JavaScript
class ChatApp {
    constructor() {
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.connectionStatus = document.getElementById('connectionStatus');
        this.statusText = document.getElementById('statusText');
        
        this.isLoading = false;
        this.messageHistory = [];
        
        this.initializeEventListeners();
        this.checkConnection();
    }

    initializeEventListeners() {
        // Send button click
        this.sendButton.addEventListener('click', () => this.sendMessage());
        
        // Enter key press
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Input validation
        this.messageInput.addEventListener('input', () => {
            this.validateInput();
        });

        // Auto-resize input (if needed)
        this.messageInput.addEventListener('input', () => {
            this.autoResizeInput();
        });
    }

    validateInput() {
        const message = this.messageInput.value.trim();
        const isValid = message.length > 0 && message.length <= 500;
        
        this.sendButton.disabled = !isValid || this.isLoading;
        
        // Update input styling based on length
        if (message.length > 450) {
            this.messageInput.classList.add('text-warning');
        } else {
            this.messageInput.classList.remove('text-warning');
        }
    }

    autoResizeInput() {
        // Reset height to auto to get correct scrollHeight
        this.messageInput.style.height = 'auto';
        
        // Set height based on content, with max height
        const maxHeight = 120; // approximately 5 lines
        const newHeight = Math.min(this.messageInput.scrollHeight, maxHeight);
        this.messageInput.style.height = newHeight + 'px';
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        
        if (!message || this.isLoading) {
            return;
        }

        // Add user message to chat
        this.addMessage(message, 'user');
        
        // Clear input
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        this.validateInput();
        
        // Show loading state
        this.setLoadingState(true);
        this.showTypingIndicator();

        try {
            const response = await this.callChatAPI(message);
            
            // Hide typing indicator
            this.hideTypingIndicator();
            
            if (response.status === 'success') {
                this.addMessage(response.response, 'bot');
            } else {
                this.addMessage(response.response || 'Sorry, I encountered an error.', 'bot', true);
            }
        } catch (error) {
            console.error('Chat error:', error);
            this.hideTypingIndicator();
            this.addMessage('Уучлаарай, сүлжээнд алдаа гарлаа та дахин оролдоно уу.', 'bot', true);
            this.showErrorToast('Connection error. Please check your network and try again.');
            this.updateConnectionStatus(false);
        } finally {
            this.setLoadingState(false);
        }
    }

    async callChatAPI(message) {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }

    addMessage(content, sender, isError = false) {
        // Remove welcome message if it exists
        const welcomeMessage = this.chatMessages.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message${isError ? ' error-message' : ''}`;
        
        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-text">${this.escapeHtml(content)}</div>
                <div class="message-time">${timestamp}</div>
            </div>
        `;

        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        
        // Store in history
        this.messageHistory.push({
            content: content,
            sender: sender,
            timestamp: new Date(),
            isError: isError
        });
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    setLoadingState(loading) {
        this.isLoading = loading;
        this.sendButton.disabled = loading || !this.messageInput.value.trim();
        
        if (loading) {
            this.sendButton.classList.add('loading');
            this.sendButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        } else {
            this.sendButton.classList.remove('loading');
            this.sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
        }
    }

    showTypingIndicator() {
        this.typingIndicator.style.display = 'block';
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        this.typingIndicator.style.display = 'none';
    }

    scrollToBottom() {
        // Use requestAnimationFrame for smooth scrolling
        requestAnimationFrame(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        });
    }

    showErrorToast(message) {
        const errorToast = document.getElementById('errorToast');
        const errorMessage = document.getElementById('errorMessage');
        
        errorMessage.textContent = message;
        
        const toast = new bootstrap.Toast(errorToast);
        toast.show();
    }

    async checkConnection() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            
            this.updateConnectionStatus(response.ok && data.status === 'healthy');
            
            if (!data.pipeline_ready) {
                this.showErrorToast('Chatbot service is initializing. Please wait a moment and try again.');
            }
        } catch (error) {
            console.error('Health check failed:', error);
            this.updateConnectionStatus(false);
        }
    }

    updateConnectionStatus(connected) {
        if (connected) {
            this.connectionStatus.style.background = 'var(--bank-success)';
            this.statusText.textContent = 'Connected';
        } else {
            this.connectionStatus.style.background = 'var(--bank-danger)';
            this.statusText.textContent = 'Disconnected';
        }
    }

    // Utility method to clear chat
    clearChat() {
        this.chatMessages.innerHTML = `
            <div class="welcome-message">
                <div class="text-center mb-4">
                    <div class="welcome-icon">
                        <i class="fas fa-robot"></i>
                    </div>
                    <h4 class="text-primary mb-2">МУХБ-ны чатботод тавтай морил</h4>
                    <p class="text-muted">Би таны асуултад хариулахад бэлэн байна.</p>
                </div>
            </div>
        `;
        this.messageHistory = [];
    }

    // Accessibility: Focus management
    focusInput() {
        this.messageInput.focus();
    }
}

// Initialize the chat application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const chatApp = new ChatApp();
    
    // Global reference for debugging (remove in production)
    window.chatApp = chatApp;
    
    // Periodic connection check
    setInterval(() => {
        chatApp.checkConnection();
    }, 30000); // Check every 30 seconds
    
    // Focus input on page load
    chatApp.focusInput();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && window.chatApp) {
        window.chatApp.checkConnection();
    }
});

// Handle online/offline events
window.addEventListener('online', () => {
    if (window.chatApp) {
        window.chatApp.checkConnection();
    }
});

window.addEventListener('offline', () => {
    if (window.chatApp) {
        window.chatApp.updateConnectionStatus(false);
        window.chatApp.showErrorToast('You are offline. Please check your internet connection.');
    }
});
