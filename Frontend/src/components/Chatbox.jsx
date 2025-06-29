import React, { useState } from 'react';

export default function Chatbox() {
  const [history, setHistory] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;
    setLoading(true);
    const updated = [...history, { role: 'user', content: input }];
    setHistory(updated);
    setInput('');

    // Mock reply (since backend API is removed)
    const reply = 'This is a mock reply for demonstration.';
    setHistory((prev) => [...prev, { role: 'assistant', content: reply }]);
    setLoading(false);
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {history.map((m, i) => (
          <div key={i} className={`message ${m.role}`}>{m.content}</div>
        ))}
      </div>
      <div className="input-area">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Type a message..."
        />
        <button onClick={sendMessage} disabled={loading}>
          {loading ? 'Generating reply...' : 'Send'}
        </button>
      </div>
    </div>
  );
}
