import { createRoot } from "react-dom/client";
import App from "./app/App";

// @ts-ignore
import "./styles/index.css";

createRoot(document.getElementById("root")!).render(<App />);