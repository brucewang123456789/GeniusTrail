import { render, fireEvent, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { vi } from 'vitest';
import React from 'react';
import ChatComponent from '@components/ChatComponent';

const mockOnMessageSend = vi.fn();

describe('ChatComponent', () => {
  beforeEach(() => {
    mockOnMessageSend.mockClear();
    mockOnMessageSend.mockResolvedValueOnce({
      success: true,
      data: { text: 'Hello, world!' },
    });
  });

  afterEach(() => {
    mockOnMessageSend.mockReset();
  });

  test('renders and handles successful response', async () => {
    render(<ChatComponent onMessageSend={mockOnMessageSend} />);

    fireEvent.change(screen.getByPlaceholderText(/Type a message/i), {
      target: { value: 'Hello!' },
    });
    fireEvent.click(screen.getByText(/Send/i));

    await waitFor(() => {
      const message = screen.queryByText('Hello, world!');
      expect(message).not.toBeNull();
    });

    expect(mockOnMessageSend).toHaveBeenCalledWith('Hello!');
    expect(screen.queryByText(/An error occurred/)).toBeNull();
  });

  test('handles server error response', async () => {

    mockOnMessageSend.mockReset();
    mockOnMessageSend.mockResolvedValueOnce({
      success: false,
      error: 'Internal Server Error',
    });

    render(<ChatComponent onMessageSend={mockOnMessageSend} />);

    fireEvent.change(screen.getByPlaceholderText(/Type a message/i), {
      target: { value: 'Hello!' },
    });
    fireEvent.click(screen.getByText(/Send/i));


    const errorElement = await screen.findByText(/An error occurred: Internal Server Error/);
    expect(errorElement).toBeInTheDocument();

    expect(mockOnMessageSend).toHaveBeenCalledWith('Hello!');
    expect(screen.queryByText('Hello, world!')).toBeNull();
  });
});