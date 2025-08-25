import React, { useEffect, useState } from "react";

const ManageUsers = () => {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchUsers = async () => {
    try {
      const res = await fetch("http://localhost:5000/admin/users");
      const data = await res.json();
      setUsers(data);
    } catch (err) {
      console.error("Error fetching users:", err);
    }
    setLoading(false);
  };

  const deleteUser = async (userId, role) => {
    if (!window.confirm(`Delete this ${role}?`)) return;

    try {
      const res = await fetch(`http://localhost:5000/admin/user/${role}/${userId}`, {
        method: "DELETE",
      });
      if (res.ok) {
        setUsers(users.filter((user) => user._id !== userId));
      } else {
        alert("Failed to delete user");
      }
    } catch (err) {
      console.error("Delete failed:", err);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  if (loading) return <p>Loading users...</p>;

  return (
    <div>
      <h2>Manage Users</h2>
      {users.length === 0 && <p>No users found.</p>}
      <table border="1" cellPadding="8" cellSpacing="0">
        <thead>
          <tr>
            <th>Username / Name</th>
            <th>Email</th>
            <th>Role</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {users.map(({ _id, username, email, role }) => (
            <tr key={_id}>
              <td>{username || "N/A"}</td>
              <td>{email}</td>
              <td>{role}</td>
              <td>
                <button onClick={() => deleteUser(_id, role)}>Delete</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ManageUsers;
