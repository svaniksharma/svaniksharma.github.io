<script lang="ts">
  let { data } = $props();
</script>

<svelte:head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
</svelte:head>

<style>
  :global(.blog-content a) {
    color: black !important;
  }
  :global(.blog-content img) {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1.5rem auto;
    border-radius: 0.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
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
<div class="container-fluid">
  <div class="row justify-content-center">
    <div class="col-xl-6 col-lg-8 col-md-10 col-12">
      <article class="blog-post">
        <h1 class="blog-title">{data.title}</h1>
        <p class="blog-date text-muted">{data.date}</p>
        <div class="blog-content">
          {@html data.markdown}
        </div>
      </article>
    </div>
  </div>
</div>
