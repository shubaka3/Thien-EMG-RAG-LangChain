(function() {
    // Load config if available
    const config = window.APP_CONFIG || {
        CHAT_IFRAME_URL: 'http://14.225.215.151:5000/chat-widget.html',
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
        overflow: hidden; /* Ensure image fits */
    `;
    
    // Use an <img> tag for the chat bubble icon
    var chatIcon = document.createElement('img');
    chatIcon.id = 'chatBubbleIcon';
    chatIcon.src = 'https://elearning.emg.edu.vn/theme/emgelementary/pix/login/logo.png'; // New icon path
    chatIcon.style.cssText = `
        width: 100%;
        height: 100%;
        object-fit: contain; /* Make sure the image scales properly */
        padding: 8px; /* Add some padding if needed */
        transition: transform 0.3s ease;
    `;
    chatButton.appendChild(chatIcon); // Append the image to the button
    
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
    chatIframe.src = config.CHAT_IFRAME_URL || 'http://14.225.215.151:5000/chat-widget.html';
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
        opacity: 0; /* Start hidden for animation */
        transform: translateY(20px); /* Start slightly off for animation */
        transition: all 0.3s ease;
    `;
    
    // Add CORS headers for iframe
    chatIframe.setAttribute('sandbox', 'allow-scripts allow-same-origin allow-forms');
    
    document.body.appendChild(chatIframe);

    // Toggle chat widget
    var isOpen = false;
    chatButton.onclick = function(event) {
        // Prevent the click event from bubbling up to the document listener immediately
        event.stopPropagation(); 

        if (!isOpen) {
            chatIframe.style.display = 'block';
            setTimeout(() => { // Small delay to allow 'display: block' to register before animation
                chatIframe.style.opacity = '1';
                chatIframe.style.transform = 'translateY(0)';
            }, 10);
            
            // Change icon to 'X'
            chatIcon.src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-x"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>';
            chatIcon.style.padding = '18px'; // Adjust padding for 'X' icon
            chatIcon.style.objectFit = 'initial'; // Reset object-fit for SVG icon
            
            isOpen = true;
        } else {
            chatIframe.style.opacity = '0';
            chatIframe.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                chatIframe.style.display = 'none';
            }, 300); // Match this with your CSS transition duration
            
            // Change icon back to the chat bubble logo
            chatIcon.src = 'https://elearning.emg.edu.vn/theme/emgelementary/pix/login/logo.png';
            chatIcon.style.padding = '8px'; // Revert padding for logo
            chatIcon.style.objectFit = 'contain'; // Revert object-fit for logo
            
            isOpen = false;
        }
    };

    // Close when clicking outside
    document.addEventListener('click', function(event) {
        // Check if the click target is NOT the iframe AND NOT the chat button (or any of its descendants)
        if (isOpen && 
            !chatIframe.contains(event.target) && 
            !chatButton.contains(event.target)) {
            
            // Programmatically click the chatButton to trigger its closing logic
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