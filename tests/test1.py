from tfitpy.ppi import shortest_path_score, hypergeom_index_score

config1 = {"data_path":"$HOME/projects/bio-datasets"}


test1 =  {
      "target_gene": "GFAP",
      "gene_cluster": "AIRE,BARX1,BAX,CEBPB,ELF4,ELK3,FOXB1,GLIS3,GSX2,HMBOX1,HOXA9,IRF3,IRF9,NFATC1,NFE2L1,NMI,PBX3,SP110,SPI1,SPR,STAT2,STAT5A,TBX19,YBX1,ZBTB7B,ZNF114,ZSCAN2",
      "n_genes": 27,
      "cluster_id": 0
    }

# s,t,m = shortest_path_score(config=config1,module=test1,ppi_source="hippie")

# print(s)
# print(t)


s1,t1,m1 = hypergeom_index_score(config=config1,module=test1,ppi_source="hippie")

print(s1)
print(t1)
print(m1)