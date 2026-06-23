<script>
  // type is the LaTeX.css class: "theorem", "lemma", "definition", "proof", "corollary"
  export let type = "theorem";
</script>

<div class={type}>
  <slot />
</div>