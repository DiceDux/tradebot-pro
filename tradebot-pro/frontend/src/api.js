import axios from "axios";
const API_BASE = "http://127.0.0.1:8000/api/bot";

export const getAccounts = () => axios.get(`${API_BASE}/accounts`);
export const addAccount = (data) => axios.post(`${API_BASE}/accounts`, data);
export const getBotStatus = () => axios.get(`${API_BASE}/status`);
export const setBotStatus = (status) => axios.post(`${API_BASE}/status?status=${status}`);