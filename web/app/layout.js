import './globals.css';

export const metadata = {
  title: 'Somnus',
  description: 'Somnus sleep strategist web app'
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
