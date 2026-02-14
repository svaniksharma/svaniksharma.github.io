import matter from 'gray-matter'

export function getPostMetadata() {
  const importedPosts = import.meta.glob("./posts/*.md", { eager: true, query: "?raw", import: "default" });
  const parsedPosts = Object.entries(importedPosts).map(([path, content]) => {
    const { data, content: markdown } = matter(content);
    return { path, frontmatter: data, markdown };
  })

  const postMeta = [];
  for (const i in parsedPosts) {
    const postData = parsedPosts[i]
    if (postData) {
      postMeta.push({ ...postData.frontmatter, markdown: postData.markdown });
    }
  }
  return postMeta;
}

export function parseDate(date) {
  const options = { year: "numeric", month: "long", day: "numeric" };
  return date.toLocaleString("en-US", options);
}
