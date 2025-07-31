(function() {
    // Load config if available
    const config = window.APP_CONFIG || {
        CHAT_IFRAME_URL: 'http://localhost:5500/chat-widget.html',
        PRIMARY_COLOR: '#0091FC'
    };

    // Create chat button
    var chatButton = document.createElement('div');
    chatButton.id = 'myChatButton';
    chatButton.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 60px;
        height: 60px;
        background: ${config.PRIMARY_COLOR};
        border-radius: 50%;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        cursor: pointer;
        z-index: 9999;
        display: flex;
        justify-content: center;
        align-items: center;
        transition: all 0.3s ease;
    `;
    
    // You can change this icon or use an image
    chatButton.innerHTML = '<span style="color:white; font-size:30px;">💬</span>';
    
    // Hover effect
    chatButton.onmouseenter = function() {
        this.style.transform = 'scale(1.1)';
        this.style.boxShadow = '0 6px 16px rgba(0,0,0,0.4)';
    };
    
    chatButton.onmouseleave = function() {
        this.style.transform = 'scale(1)';
        this.style.boxShadow = '0 4px 12px rgba(0,0,0,0.3)';
    };

    document.body.appendChild(chatButton);

    // Create iframe for chat widget
    var chatIframe = document.createElement('iframe');
    chatIframe.id = 'myChatIframe';
    chatIframe.src = config.CHAT_IFRAME_URL || 'http://localhost:5500/chat-widget.html';
    chatIframe.style.cssText = `
        position: fixed;
        bottom: 90px;
        right: 20px;
        width: 350px;
        height: 500px;
        border: none;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        z-index: 9999;
        display: none;
        transition: all 0.3s ease;
    `;
    
    // Add CORS headers for iframe
    chatIframe.setAttribute('sandbox', 'allow-scripts allow-same-origin allow-forms');
    
    document.body.appendChild(chatIframe);

    // Toggle chat widget
    var isOpen = false;
    chatButton.onclick = function() {
        if (!isOpen) {
            chatIframe.style.display = 'block';
            chatIframe.style.opacity = '0';
            chatIframe.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                chatIframe.style.opacity = '1';
                chatIframe.style.transform = 'translateY(0)';
            }, 10);
            
            chatButton.innerHTML = '<span style="color:white; font-size:24px;">✕</span>';
            isOpen = true;
        } else {
            chatIframe.style.opacity = '0';
            chatIframe.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                chatIframe.style.display = 'none';
            }, 300);
            
            chatButton.innerHTML = '<span style="color:white; font-size:30px;">💬</span>';
            isOpen = false;
        }
    };

    // Close when clicking outside
    document.addEventListener('click', function(event) {
        if (isOpen && 
            !chatIframe.contains(event.target) && 
            !chatButton.contains(event.target)) {
            chatButton.click();
        }
    });

    // Responsive design
    function adjustForMobile() {
        if (window.innerWidth <= 768) {
            chatIframe.style.width = 'calc(100vw - 40px)';
            chatIframe.style.height = 'calc(100vh - 140px)';
            chatIframe.style.right = '20px';
            chatIframe.style.left = '20px';
            chatIframe.style.bottom = '90px';
        } else {
            chatIframe.style.width = '350px';
            chatIframe.style.height = '500px';
            chatIframe.style.right = '20px';
            chatIframe.style.left = 'auto';
            chatIframe.style.bottom = '90px';
        }
    }

    window.addEventListener('resize', adjustForMobile);
    adjustForMobile();

    console.log('Chat widget loaded successfully!');
})();