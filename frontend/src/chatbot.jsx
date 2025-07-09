// src/chatbot.jsx
import { useState } from 'react'
import axios from 'axios'

function ChatBox({ user }) {
  const [input, setInput] = useState("")
  const [messages, setMessages] = useState([])

  const sendMessage = async () => {
    if (!input.trim()) return;

    try {
      const res = await axios.post("http://localhost:8000/chat", {
        user_input: input,
        thread_id: user.id  // ✅ Use Supabase user ID as thread_id
      })

      const response = res.data.response

      setMessages(prev => [...prev, { user: input, bot: response }])
      setInput("")
    } catch (err) {
      console.error("Error:", err)
      setMessages(prev => [...prev, { user: input, bot: "❌ Failed to get response" }])
    }
  }

  return (
    <div>
      <div>
        {messages.map((msg, idx) => (
          <div key={idx} style={{ marginBottom: "1rem" }}>
            <p><strong>You:</strong> {msg.user}</p>
            <p><strong>Bot:</strong> {msg.bot}</p>
          </div>
        ))}
      </div>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Ask something..."
        style={{ width: "300px", marginRight: "10px" }}
      />
      <button onClick={sendMessage}>Send</button>
    </div>
  )
}

export default ChatBox
