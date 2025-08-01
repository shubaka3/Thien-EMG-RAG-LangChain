// config.js - Cấu hình toàn cục cho ứng dụng chat
window.APP_CONFIG = {
    // API Endpoints
    CHAT_API_URL: 'http://vmentor-service.emg.edu.vn/api/chat/completion',
    CONFIG_GET_URL: 'http://vmentor-service.emg.edu.vn/api/config/get_dynamic_vars',
    CONFIG_UPDATE_URL: 'http://vmentor-service.emg.edu.vn/api/config/update',
    
    // Widget Settings
    CHAT_IFRAME_URL: 'http://vmentor.emg.edu.vn/chat-widget.html',
    
    // Avatar URLs
    AVATAR_BOT_URL: 'https://digital.com.vn/wp-content/uploads/2025/03/ai-bot.jpg',
    AVATAR_USER_URL: 'https://static.vecteezy.com/system/resources/thumbnails/004/607/791/small_2x/man-face-emotive-icon-smiling-male-character-in-blue-shirt-flat-illustration-isolated-on-white-happy-human-psychological-portrait-positive-emotions-user-avatar-for-app-web-design-vector.jpg',
    // thêm để update
    // UI Colors
    PRIMARY_COLOR: '#0091FC',
    SECONDARY_COLOR: '#f8f9fa',
    SUCCESS_COLOR: '#28a745',
    DANGER_COLOR: '#dc3545',
    WARNING_COLOR: '#ffc107',
    INFO_COLOR: '#17a2b8',
    
    // Support Information
    SUPPORT_NAME: 'Huỳnh Phú Thiện',
    SUPPORT_EMAIL: 'thienhp@emg.vn',
    
    // Language Settings
    DEFAULT_LANGUAGE: 'Tiếng Việt',
    FLAG_BASE_URL: 'https://flagcdn.com/w40',
    
    // Chat Settings
    MAX_MESSAGE_LENGTH: 4000,
    TYPING_DELAY: 1000,
    AUTO_SCROLL: true,
    
    // Widget Position (for embed.js)
    WIDGET_POSITION: {
        bottom: '20px',
        right: '20px'
    },
    
    // Widget Size
    WIDGET_SIZE: {
        width: '350px',
        height: '500px',
        buttonSize: '60px'
    },
    
    // Features
    FEATURES: {
        MARKDOWN_SUPPORT: true,
        CODE_HIGHLIGHTING: true,
        IMAGE_SUPPORT: true,
        PDF_SUPPORT: true,
        COPY_CODE: true,
        LANGUAGE_SELECTOR: true,
        TYPING_INDICATOR: true
    },
    
    // CORS Settings
    CORS: {
        ALLOW_ALL_ORIGINS: true,
        ALLOWED_ORIGINS: ['*'],
        ALLOWED_METHODS: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
        ALLOWED_HEADERS: ['Content-Type', 'Authorization', 'X-Requested-With']
    }
};

// Utility functions
window.APP_UTILS = {
    // Format message content
    formatMessage: function(content) {
        if (!content) return '';
        
        // Basic HTML escaping
        content = content.replace(/&/g, '&amp;')
                        .replace(/</g, '&lt;')
                        .replace(/>/g, '&gt;')
                        .replace(/"/g, '&quot;')
                        .replace(/'/g, '&#39;');
        
        // Convert URLs to links
        content = content.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank">$1</a>');
        
        return content;
    },
    
    // Get current timestamp
    getCurrentTimestamp: function() {
        return new Date().toISOString();
    },
    
    // Generate unique ID
    generateId: function() {
        return 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    },
    
    // Check if mobile device
    isMobile: function() {
        return window.innerWidth <= 768;
    },
    
    // Debounce function
    debounce: function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
};

// Export for Node.js if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { APP_CONFIG: window.APP_CONFIG, APP_UTILS: window.APP_UTILS };
}