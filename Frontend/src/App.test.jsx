import '@testing-library/jest-dom';
import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import App from './App';

test('renders learn react link', () => {
  render(<App />);
  const linkElement = screen.getByText(/Veltraxor/i);
  expect(linkElement).toBeInTheDocument();  // 使用 @testing-library/jest-dom 扩展
});
