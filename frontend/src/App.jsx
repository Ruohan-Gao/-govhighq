// src/App.jsx
import { useEffect, useState } from 'react'
import { supabase } from './supabaseClient'
import Login from './Login'
import ChatBox from './chatbot'
import LogoutButton from './LogoutButton'  // ✅ Import it

function App() {
  const [user, setUser] = useState(null)

  useEffect(() => {
    supabase.auth.getUser().then(({ data: { user } }) => {
      setUser(user)
    })

    supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user || null)
    })
  }, [])

  if (!user) return <Login />

  return (
    <div>
      <h1>🔒 Authenticated Chat with LLM</h1>
      <LogoutButton />  ✅ Show logout button
      <ChatBox user={user} />
    </div>
  )
}

export default App
