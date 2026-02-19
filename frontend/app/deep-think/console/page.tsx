import { redirect } from "next/navigation";

// Dedicated route for engineering console; keep URL separation while reusing the same page implementation.
export default function DeepThinkConsolePage() {
  redirect("/deep-think?workspace=console");
}
