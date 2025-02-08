// JavaScript para manejar el envío de mensajes
document.getElementById('send-button').addEventListener('click', function() {
    var messageInput = document.getElementById('message-input');
    var message = messageInput.value;
    messageInput.value = '';

    // Añadir el mensaje del usuario a la interfaz
    var chatHistory = document.querySelector('.chat-history ul');
    var userMessage = '<li class="clearfix">' +
        '<div class="message-data text-right">' +
        '<span class="message-data-time">Ahora</span>' +
        '<img src="https://bootdey.com/img/Content/avatar/avatar7.png" alt="avatar">' +
        '</div>' +
        '<div class="message other-message float-right">' + message + '</div>' +
        '</li>';
    chatHistory.innerHTML += userMessage;

    // Enviar el mensaje al servidor
    fetch("{% url 'chatbot' %}", {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
            'X-CSRFToken': '{{ csrf_token }}'
        },
        body: 'message=' + encodeURIComponent(message)
    })
    .then(response => response.json())
    .then(data => {
        // Añadir la respuesta del asistente a la interfaz
        var assistantMessage = '<li class="clearfix">' +
            '<div class="message-data">' +
            '<span class="message-data-time">Ahora</span>' +
            '</div>' +
            '<div class="message my-message">' + data.response + '</div>' +
            '</li>';
        chatHistory.innerHTML += assistantMessage;

        // Desplazar hacia abajo
        chatHistory.scrollTop = chatHistory.scrollHeight;
    })
    .catch(error => console.error('Error:', error));
});
