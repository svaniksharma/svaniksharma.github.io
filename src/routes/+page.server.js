import sanitizeHtml from "sanitize-html";
import sbd from "sbd";
import { getHtmlPostMetadata, parseDate } from "./utilities.js";

const postMeta = getHtmlPostMetadata();

function processText(str, maxLength = 8000) {
    const sentences = sbd.sentences(str, {
        newline_boundaries: false,
        html_boundaries: false,
        sanitize: false,
    });
    let result = sentences.slice(0, 2).join(" ");
    if (result.length > maxLength) {
        result = result.substring(0, maxLength).trim() + "...";
    }

    return result;
}

function getSummary(post) {
    if (post.isMarkdown) {
        // TODO
    } else {
        return processText(
            sanitizeHtml(post.postContent, {
                allowedTags: [],
                allowedAttributes: {},
            }),
        );
    }
}

export function load() {
    return {
        summaries: postMeta
            .sort((a, b) => (a.date > b.date ? -1 : a.date < b.date ? 1 : 0))
            .map((post) => ({
                title: post.title,
                slug: post.slug,
                date: parseDate(post.date),
                summary: getSummary(post),
            })),
    };
}
