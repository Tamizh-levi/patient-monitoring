// Navbar.jsx
import React from "react";
import { Link, useNavigate } from "react-router-dom";
import "./Navbar.css";

const Navbar = () => {
  const navigate = useNavigate();
  const role = localStorage.getItem("role");
  const userEmail = localStorage.getItem("email");

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("role");
    localStorage.removeItem("email");
    navigate("/login");
  };

  // Get color based on role
  const getNavbarColor = () => {
    const roleColors = {
      patient: "var(--patient-color)",
      doctor: "var(--doctor-color)",
      pharmacy: "var(--pharmacy-color)",
      admin: "var(--admin-color)",
      default: "var(--default-color)"
    };
    return roleColors[role] || roleColors.default;
  };

  return (
    <nav className="navbar" style={{ backgroundColor: getNavbarColor() }}>
      <div className="brand">
        <Link to="/" className="brand-link">🏥 CluCare</Link>
      </div>

      <div className="nav-content">
        {userEmail && (
          <div className="user-info">
            <span className="user-role">{role?.toUpperCase()}</span>
            <span className="user-email">{userEmail}</span>
          </div>
        )}

        <div className="nav-links">
          {role === "patient" && (
            <>
              <Link to="/patient/dashboard" className="nav-link">Dashboard</Link>
              <Link to="/patient/appointments" className="nav-link">Appointments</Link>
              <Link to="/patient/prescriptions" className="nav-link">Prescriptions</Link>
            </>
          )}
          {role === "doctor" && (
            <>
              <Link to="/doctor/dashboard" className="nav-link">Dashboard</Link>
              <Link to="/doctor/schedule" className="nav-link">Schedule</Link>
              <Link to="/doctor/patients" className="nav-link">Patients</Link>
            </>
          )}
          {role === "pharmacy" && (
            <>
              <Link to="/pharmacy/dashboard" className="nav-link">Dashboard</Link>
              <Link to="/pharmacy/orders" className="nav-link">Orders</Link>
              <Link to="/pharmacy/inventory" className="nav-link">Inventory</Link>
            </>
          )}
          {role === "admin" && (
            <>
              <Link to="/admin/dashboard" className="nav-link">Dashboard</Link>
              <Link to="/admin/users" className="nav-link">Users</Link>
              <Link to="/admin/reports" className="nav-link">Reports</Link>
            </>
          )}
          
          {role ? (
            <button onClick={handleLogout} className="logout-btn">
              Logout
            </button>
          ) : (
            <Link to="/" className="login-link">Login / Register</Link>
          )}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;