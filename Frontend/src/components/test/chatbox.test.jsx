import 'vitest-dom/extend-expect';
import React from 'react';
import { render, screen } from '@testing-library/react';
import Chatbox from '@components/Chatbox';

describe('<Chatbox /> smoke test', () => {
  it('renders the input field and send button', () => {
    render(<Chatbox />);
    // check that the input field is rendered
    expect(screen.getByPlaceholderText('Type a message...')).toBeInTheDocument();
    // check that the send button is rendered
    expect(screen.getByRole('button', { name: 'Send' })).toBeInTheDocument();
  });
});
