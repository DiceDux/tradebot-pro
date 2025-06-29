import React, { useEffect, useState } from "react";
import { getBotStatus, setBotStatus } from "../api";

export default function BotControl() {
  const [status, setStatus] = useState("unknown");

  useEffect(() => {
    getBotStatus().then(res => setStatus(res.data.status));
  }, []);

  const handleChange = async (newStatus) => {
    await setBotStatus(newStatus);
    setStatus(newStatus);
  };

  return (
    <div>
      <h2>وضعیت ربات: {status}</h2>
      <button onClick={() => handleChange("running")}>شروع</button>
      <button onClick={() => handleChange("stopped")}>توقف</button>
      <button onClick={() => handleChange("sleeping")}>خواب</button>
    </div>
  );
}