import { getHtmlPostMetadata, parseDate } from "../../utilities";
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkMath from "remark-math";
import remarkRehype from "remark-rehype";
import rehypeMathjax from "rehype-mathjax";
import rehypeHighlight from "rehype-highlight";
import rehypeStringify from "rehype-stringify";

export const prerender = true;

const postMeta = getHtmlPostMetadata();

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

export async function load({ params }) {
    const post = postMeta.find((post) => post.slug == params.slug);
    //const processedMarkdown = await processMarkdown(post.markdown);
    return {
        date: parseDate(post.date),
        title: post.title,
        postContent: post.postContent,
    };
}
