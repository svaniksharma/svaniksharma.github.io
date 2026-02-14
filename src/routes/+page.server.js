import removeMd from 'remove-markdown'
import sbd from 'sbd'
import { getPostMetadata, parseDate } from './utilities.js'

const postMeta = getPostMetadata("./posts/*.md");

function processText(str, maxLength = 8000) {
  const sentences = sbd.sentences(str, {
    newline_boundaries: false,
    html_boundaries: false,
    sanitize: false
  });
  let result = sentences.slice(0, 2).join(' ');
  if (result.length > maxLength) {
    result = result.substring(0, maxLength).trim() + '...';
  }

  return result;
}

export function load() {
  return {
    summaries: postMeta
      .sort((a, b) => a.date > b.date ? -1 : (a.date < b.date ? 1 : 0))
      .map((post) => ({
        title: post.title,
        slug: post.slug,
        date: parseDate(post.date),
        summary: processText(removeMd(post.markdown))
      }))
  }
}
