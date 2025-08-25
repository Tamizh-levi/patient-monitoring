import React, { useState } from "react";
import ManageUsers from "./ManageUsers";
import ViewAllAppointments from "./ViewAllAppointments";
import Reports from "./Reports";

const AdminDashboard = () => {
  const [page, setPage] = useState("users");

  return (
    <div>
      <h1>Admin Dashboard</h1>
      <nav>
        <button onClick={() => setPage("users")}>Manage Users</button>
        <button onClick={() => setPage("appointments")}>View Appointments</button>
        <button onClick={() => setPage("reports")}>Reports</button>
      </nav>

      <section>
        {page === "users" && <ManageUsers />}
        {page === "appointments" && <ViewAllAppointments />}
        {page === "reports" && <Reports />}
      </section>
    </div>
  );
};

export default AdminDashboard;
