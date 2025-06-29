import React, { useEffect, useState } from "react";
import { getAccounts, addAccount } from "../api";

export default function Accounts() {
  const [accounts, setAccounts] = useState([]);
  const [form, setForm] = useState({ name: "", exchange: "", api_key: "", api_secret: "" });

  useEffect(() => {
    getAccounts().then(res => setAccounts(res.data));
  }, []);

  const handleChange = (e) => setForm({ ...form, [e.target.name]: e.target.value });

  const handleSubmit = async (e) => {
    e.preventDefault();
    await addAccount(form);
    const res = await getAccounts();
    setAccounts(res.data);
    setForm({ name: "", exchange: "", api_key: "", api_secret: "" });
  };

  return (
    <div>
      <h2>لیست حساب‌ها</h2>
      <ul>
        {accounts.map(acc => (
          <li key={acc.id}>{acc.name} | {acc.exchange} | {acc.is_active ? "فعال" : "غیرفعال"}</li>
        ))}
      </ul>
      <h3>افزودن حساب جدید</h3>
      <form onSubmit={handleSubmit}>
        <input name="name" placeholder="نام" value={form.name} onChange={handleChange} required />
        <input name="exchange" placeholder="صرافی" value={form.exchange} onChange={handleChange} required />
        <input name="api_key" placeholder="API Key" value={form.api_key} onChange={handleChange} required />
        <input name="api_secret" placeholder="API Secret" value={form.api_secret} onChange={handleChange} required />
        <button type="submit">افزودن</button>
      </form>
    </div>
  );
}