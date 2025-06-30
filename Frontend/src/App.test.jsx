import 'vitest-dom/extend-expect';
import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders learn react link', () => {
  render(<App />);
  const linkElement = screen.getByText(/Veltraxor/i);
  expect(linkElement).toBeInTheDocument();
});
