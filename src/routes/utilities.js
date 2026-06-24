import matter from "gray-matter";

export function getHtmlPostMetadata() {
    const htmlPosts = import.meta.glob("./posts/*.html", {
        eager: true,
        query: "?raw",
        import: "default",
    });
    return getPostMetadata(htmlPosts);
}

export function getMarkdownPostMetadata() {
    const markdownPosts = import.meta.glob("./posts/*.md", {
        eager: true,
        query: "?raw",
        import: "default",
    });
    return getPostMetadata(markdownPosts);
}

function getPostMetadata(posts) {
    const parsedPosts = Object.entries(posts).map(([path, content]) => {
        const { data, content: postContent } = matter(content);
        return { path, frontmatter: data, postContent };
    });
    const postMeta = [];
    for (const i in parsedPosts) {
        const postData = parsedPosts[i];
        if (postData) {
            postMeta.push({
                ...postData.frontmatter,
                postContent: postData.postContent,
            });
        }
    }
    return postMeta;
}

export function parseDate(date) {
    const options = { year: "numeric", month: "long", day: "numeric" };
    return date.toLocaleString("en-US", options);
}
