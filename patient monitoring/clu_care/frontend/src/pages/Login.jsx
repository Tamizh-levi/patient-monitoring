// src/pages/Login.jsx
import React, { useState } from "react";
import "./Login.css";
import logo from '../assets/logo.png'; // Adjusted relative path

function Login() {
  const [role, setRole] = useState("patient");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleLogin = (e) => {
    e.preventDefault();
    console.log({ role, email, password });

fetch('http://localhost:5000/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ role, email, password })
})

      .then((res) => res.json())
      .then((data) => {
        console.log("Login Success:", data);
        if (role === "patient") window.location.href = "/patient/dashboard";
        if (role === "doctor") window.location.href = "/doctor/dashboard";
        if (role === "admin") window.location.href = "/admin/dashboard";
        if (role === "pharmacy") window.location.href = "/pharmacy/dashboard";
      })
      .catch((err) => console.error("Login Error:", err));
  };

  // Role colors mapping
  const roleColors = {
    patient: "#2E86C1", // Calm Blue
    doctor: "#1ABC9C",  // Teal Green
    admin: "#F5B041",   // Soft Orange
    pharmacy: "#28B463" // Fresh Green
  };

  return (
    <div className="login-container">
      <div className="login-branding">
        <img src={logo} alt="CluCare Logo" className="login-logo" />
        <div className="login-header">
          <h2>Welcome to CluCare</h2>
          <p>Please login to access your account</p>
        </div>
      </div>
      
      <form onSubmit={handleLogin}>
        <div className="role-selection">
          <label>I am a:</label>
          <div className="role-options">
            {["patient", "doctor", "admin", "pharmacy"].map((r) => (
              <label 
                key={r} 
                className="role-option"
                style={{ 
                  backgroundColor: role === r ? roleColors[r] : "#F8F9F9",
                  borderColor: roleColors[r]
                }}
              >
                <input
                  type="radio"
                  name="role"
                  value={r}
                  checked={role === r}
                  onChange={() => setRole(r)}
                />
                <span>{r.charAt(0).toUpperCase() + r.slice(1)}</span>
              </label>
            ))}
          </div>
        </div>

        <div className="form-group">
          <label>Email Address</label>
          <input
            type="email"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>

        <div className="form-group">
          <label>Password</label>
          <input
            type="password"
            placeholder="Enter your password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>

        <button type="submit" className="login-button">
          Login
        </button>
      </form>
    </div>
  );
}

export default Login;