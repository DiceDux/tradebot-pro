import React from "react";
import Accounts from "./pages/Accounts";
import BotControl from "./pages/BotControl";

function App() {
  return (
    <div style={{ maxWidth: 600, margin: "auto", padding: 24 }}>
      <h1>TradeBot Pro Panel</h1>
      <BotControl />
      <Accounts />
    </div>
  );
}
export default App;