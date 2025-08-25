// src/App.js
import React from "react";
import { BrowserRouter as Router, Route, Routes, useLocation } from "react-router-dom";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import Login from "./pages/Login";
import AdminDashboard from "./pages/admin/AdminDashboard";
function LayoutWrapper({ children }) {
  const location = useLocation();
  const noNavRoutes = ["/"]; // routes where navbar/footer is hidden
  const hideNav = noNavRoutes.includes(location.pathname);

  return (
    <>
      {!hideNav && <Navbar />}
      {children}
      {!hideNav && <Footer />}
    </>
  );
}

function App() {
  return (
    <Router>
      <LayoutWrapper>
        <Routes>
          <Route path="/" element={<Login />} />
          <Route path="/admin/dashboard" element={<AdminDashboard />} />
        
        </Routes>
      </LayoutWrapper>
    </Router>
  );
}

export default App;
