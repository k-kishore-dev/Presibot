function appendMessage(message, sender) {
    const chatContainer = document.getElementById('chat-container');
    const messageElement = document.createElement('div');
    messageElement.classList.add('message');
    const senderElement = document.createElement('div');
    senderElement.classList.add('sender-' + sender.toLowerCase());
    senderElement.textContent = sender;
    messageElement.appendChild(senderElement);
    const contentElement = document.createElement('div');
    contentElement.classList.add('message-content');
    contentElement.textContent = message;
    messageElement.appendChild(contentElement);
    chatContainer.appendChild(messageElement);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function sendMessage() {
    const messageInput = document.getElementById('message');
    const message = messageInput.value.trim();
    if (!message) return;
    appendMessage(message, 'You');
    messageInput.value = '';

    // Send message to server
    fetch('/chat', {
        method: 'POST',
        body: JSON.stringify({ message: message }),
        headers: { 'Content-Type': 'application/json' }
    })
    .then(response => response.json())
    .then(data => {
        const botResponse = data.response;
        appendMessage(botResponse, 'Bot');
    })
    .catch(error => console.error('Error:', error));
}

document.getElementById('message').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});
