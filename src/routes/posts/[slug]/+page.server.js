import { getPostMetadata, parseDate } from "../../utilities";
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import rehypeStringify from 'rehype-stringify';

export const prerender = true;

const postMeta = getPostMetadata();

async function processMarkdown(markdown) {
  const result = await unified()
    .use(remarkParse)
    .use(remarkMath)
    .use(remarkRehype)
    .use(rehypeKatex)
    .use(rehypeHighlight)
    .use(rehypeStringify)
    .process(markdown);
  return String(result);
}

export async function load({ params }) {
  const post = postMeta.find((post) =>
    post.slug == params.slug
  );
  return {
    date: parseDate(post.date),
    title: post.title,
    markdown: await processMarkdown(post.markdown)
  };
}
