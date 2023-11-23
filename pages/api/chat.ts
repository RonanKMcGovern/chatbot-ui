import { DEFAULT_SYSTEM_PROMPT, DEFAULT_TEMPERATURE } from '@/utils/app/const';
import { OpenAIError, OpenAIStream } from '@/utils/server';

import { ChatBody, Message } from '@/types/chat';

// @ts-expect-error
import wasm from '../../node_modules/@dqbd/tiktoken/lite/tiktoken_bg.wasm?module';

import tiktokenModel from '@dqbd/tiktoken/encoders/cl100k_base.json';
import { Tiktoken, init } from '@dqbd/tiktoken/lite/init';

export const config = {
  runtime: 'edge',
};

const handler = async (req: Request): Promise<Response> => {
  try {
    const { model, messages, key, prompt, temperature } = (await req.json()) as ChatBody;

    await init((imports) => WebAssembly.instantiate(wasm, imports));
    const encoding = new Tiktoken(
      tiktokenModel.bpe_ranks,
      tiktokenModel.special_tokens,
      tiktokenModel.pat_str,
    );

    let promptToSend = prompt;
    if (!promptToSend) {
      promptToSend = DEFAULT_SYSTEM_PROMPT;
    }

    let temperatureToUse = temperature;
    if (temperatureToUse == null) {
      temperatureToUse = DEFAULT_TEMPERATURE;
    }

    const prompt_tokens = encoding.encode(promptToSend);

    let tokenCount = prompt_tokens.length;
    let messagesToSend: Message[] = [];

    // Assuming 4 characters per token
    const averageCharsPerToken = 4;

    for (let i = messages.length - 1; i >= 0; i--) {
      const message = messages[i];
      let tokens = encoding.encode(message.content);
      tokenCount += tokens.length;

      if (tokenCount + 1000 > model.tokenLimit) {
        // Calculate the number of tokens to trim
        let excessTokens = tokenCount + 1000 - model.tokenLimit;
        // Estimate the number of characters to trim
        let charsToTrim = excessTokens * averageCharsPerToken;
        
        if (message.content.length > charsToTrim) {
          // Trim the message content
          message.content = message.content.slice(0, -charsToTrim);
          // Re-encode to get the updated token count
          tokens = encoding.encode(message.content);
          tokenCount = prompt_tokens.length + tokens.length;
        } else {
          // If the entire message needs to be trimmed, remove the message
          messagesToSend.splice(i, 1);
          tokenCount -= tokens.length;  // Adjust the token count
        }
      }

      // If the message is not removed, add it to the messages to send
      if (messagesToSend.indexOf(message) === -1) {
        messagesToSend = [message, ...messagesToSend];
      }
    }

    encoding.free();

    const stream = await OpenAIStream(model, promptToSend, temperatureToUse, key, messagesToSend);

    return new Response(stream);
  } catch (error) {
    console.error(error);
    if (error instanceof OpenAIError) {
      return new Response('Error', { status: 500, statusText: error.message });
    } else {
      return new Response('Error', { status: 500 });
    }
  }
};

export default handler;