import React from 'react';
import { Navbar } from './Navbar';
import { Footer } from './Footer';

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <Navbar />
      <main style={{
        flex: 1,
        maxWidth: 1280,
        margin: '0 auto',
        width: '100%',
        padding: '28px 24px 48px',
      }}>
        {children}
      </main>
      <Footer />
    </div>
  );
}
