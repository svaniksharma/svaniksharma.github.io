import {
    getHtmlPostMetadata,
    getMarkdownPostMetadata,
    parseDate,
} from "../../utilities";
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkMath from "remark-math";
import remarkRehype from "remark-rehype";
import rehypeParse from "rehype-parse";
import rehypeMathjax from "rehype-mathjax";
import rehypeHighlight from "rehype-highlight";
import rehypeStringify from "rehype-stringify";

export const prerender = true;

let postMeta = getHtmlPostMetadata();
postMeta = postMeta.concat(getMarkdownPostMetadata());

async function processMarkdown(markdown) {
    const result = await unified()
        .use(remarkParse)
        .use(remarkMath)
        .use(remarkRehype, { allowDangerousHtml: true })
        .use(rehypeMathjax, { tex: { tags: "ams" } })
        .use(rehypeHighlight)
        .use(rehypeStringify, { allowDangerousHtml: true })
        .process(markdown);
    return String(result);
}

async function processHTML(htmlContent) {
    const result = await unified()
        .use(rehypeParse)
        .use(rehypeHighlight)
        .use(rehypeStringify, { allowedDangerousHtml: true })
        .process(htmlContent);
    return String(result);
}

export async function load({ params }) {
    const post = postMeta.find((post) => post.slug == params.slug);
    const processedContent = !post.isMarkdown
        ? await processHTML(post.postContent)
        : await processMarkdown(post.postContent);
    return {
        date: parseDate(post.date),
        title: post.title,
        postContent: processedContent,
    };
}
