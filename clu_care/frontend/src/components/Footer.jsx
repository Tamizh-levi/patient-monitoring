import React from "react";

const Footer = () => {
  return (
    <footer style={styles.footer}>
      <p>© {new Date().getFullYear()} E-Med Connect. All rights reserved.</p>
    </footer>
  );
};

const styles = {
  footer: {
    backgroundColor: "#264653",
    color: "#fff",
    textAlign: "center",
    padding: "10px 0",
    position: "fixed",
    bottom: 0,
    width: "100%",
  },
};

export default Footer;
