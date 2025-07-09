// src/Login.jsx
import { useState } from 'react'
import { supabase } from './supabaseClient'

function Login({ onLogin }) {
  const [email, setEmail] = useState("")

  const handleLogin = async () => {
    const { data, error } = await supabase.auth.signInWithOtp({ email })
    if (error) alert(error.message)
    else alert("Check your email for the login link!")
  }

  return (
    <div>
      <h2>Login with Email</h2>
      <input
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="you@example.com"
      />
      <button onClick={handleLogin}>Send Login Link</button>
    </div>
  )
  
}

export default Login
