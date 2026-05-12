import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const AA_PATTERN = /^[ACDEFGHIKLMNPQRSTVWYX]+$/i;
export function sanitizeSequence(value: string) {
  return value.toUpperCase().replace(/[^ACDEFGHIKLMNPQRSTVWYX]/g, "").slice(0, 30);
}
