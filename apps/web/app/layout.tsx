import type { Metadata } from "next";
import { Archivo, Instrument_Sans, Geist_Mono } from "next/font/google";
import "./globals.css";

const archivo = Archivo({
  subsets: ["latin"],
  variable: "--font-archivo",
  axes: ["wdth"],
});
const instrument = Instrument_Sans({ subsets: ["latin"], variable: "--font-instrument" });
const geistMono = Geist_Mono({ subsets: ["latin"], variable: "--font-geist-mono" });

export const metadata: Metadata = {
  title: "XG Alonso",
  description: "What to do this gameweek, and why.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${archivo.variable} ${instrument.variable} ${geistMono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
