import { supabase } from './supabaseClient'

function LogoutButton() {
  const handleLogout = async () => {
    await supabase.auth.signOut()
    window.location.reload() // 🔄 Refresh to trigger login screen
  }

  return (
    <button onClick={handleLogout} style={{ marginTop: '1rem' }}>
      🔓 Logout
    </button>
  )
}

export default LogoutButton
