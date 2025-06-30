import React, { useState } from 'react';

const ChatComponent = ({ onMessageSend }) => {
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState('');
  const [error, setError] = useState(null);

  const handleMessageChange = (e) => {
    setMessage(e.target.value);
  };

  const handleSend = async () => {
    if (onMessageSend) {
      try {
        const res = await onMessageSend(message);
        if (res.success) {
          setResponse(res.data.text);
          setError(null);
        } else {
          setError(res.error);
        }
      } catch (err) {
        setError('An unexpected error occurred');
      }
    }
    setMessage('');
  };

  return (
    <div>
      <h1>Chat Component</h1>
      <input
        type="text"
        value={message}
        onChange={handleMessageChange}
        placeholder="Type a message"
      />
      <button onClick={handleSend}>Send</button>
      {response && <div>{response}</div>}
      {error && <div>An error occurred: {error}</div>}
    </div>
  );
};

export default ChatComponent;