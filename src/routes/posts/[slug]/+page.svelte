<script lang="ts">
  let { data } = $props();
</script>

<svelte:head>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
  <link rel="stylesheet" href="https://latex.vercel.app/style.css">
  <script>
   MathJax = {
	   tex: {
		   inlineMath: {'[+]': [['$', '$']]}
	   }
   };
  </script>
  <script id="MathJax-script" src="/MathJax-4.1.2/tex-mml-chtml.js"></script>
</svelte:head>

<style>
  :global(.blog-content a) {
    color: black !important;
  }

  .blog-post {
    padding: 2rem 1rem;
  }
  
  .blog-title {
    margin-bottom: 0.25rem;
    font-weight: 700;
  }
  
  .blog-date {
    margin-bottom: 1.5rem;
    font-size: 0.95rem;
  }
</style>
<h1 class="blog-title">{data.title}</h1>
<p class="blog-date text-muted">{data.date}</p>
{@html data.postContent}
