(mod ()
  (include sha256tree.clib)
  (compile-file secret-number secret_number.cl)
  (defconst A (sha256 1 H))
  (defconst H (sha256tree secret-number))
  (defconst I (sha256 1 H))
  H
  )
