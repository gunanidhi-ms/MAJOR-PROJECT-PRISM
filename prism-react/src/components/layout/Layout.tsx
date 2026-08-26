import React from 'react';
import { Navbar } from './Navbar';
import { Footer } from './Footer';

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="app-root-layout">
      <Navbar />
      <main className="app-main-content">
        {children}
      </main>
      <Footer />
    </div>
  );
}
