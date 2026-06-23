// mdsvex.config.js
import remarkMath from "remark-math";
import rehypeKatexSvelte from "rehype-katex-svelte";
import remarkDirective from "remark-directive";
import { visit } from "unist-util-visit";
import { h } from "hastscript";

export default {
    extensions: [".svx", ".md"],
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatexSvelte],
};
