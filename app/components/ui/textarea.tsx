import * as React from "react";
import { cn } from "@/lib/utils";

const Textarea = React.forwardRef<HTMLTextAreaElement, React.TextareaHTMLAttributes<HTMLTextAreaElement>>(({ className, ...props }, ref) => (
  <textarea
    ref={ref}
    className={cn("min-h-[128px] w-full rounded-2xl border border-teal-200/20 bg-slate-950/50 px-4 py-3 text-base text-teal-50 placeholder:text-slate-500 outline-none transition focus:border-teal-300/70 focus:ring-2 focus:ring-teal-300/20", className)}
    {...props}
  />
));
Textarea.displayName = "Textarea";

export { Textarea };
