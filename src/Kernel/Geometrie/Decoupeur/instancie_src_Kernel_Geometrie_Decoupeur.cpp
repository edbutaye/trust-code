//
// Warning : DO NOT EDIT !
// To update this file, run: make depend
//
#include <verifie_pere.h>
#include <Decouper.h>
#include <Decouper_multi.h>
#include <Partitionneur_Fichier_Decoupage.h>
#include <Partitionneur_Metis.h>
#include <Partitionneur_Parmetis.h>
#include <Partitionneur_Partition.h>
#include <Partitionneur_Ptscotch.h>
#include <Partitionneur_Sous_Domaine.h>
#include <Partitionneur_Sous_Zones.h>
#include <Partitionneur_Tranche.h>
#include <Partitionneur_Union.h>
void instancie_src_Kernel_Geometrie_Decoupeur() {
Cerr << "src_Kernel_Geometrie_Decoupeur" << finl;
Decouper inst1;verifie_pere(inst1);
Decouper_multi inst2;verifie_pere(inst2);
Partitionneur_Fichier_Decoupage inst3;verifie_pere(inst3);
Partitionneur_Metis inst4;verifie_pere(inst4);
Partitionneur_Parmetis inst5;verifie_pere(inst5);
Partitionneur_Partition inst6;verifie_pere(inst6);
Partitionneur_Ptscotch inst7;verifie_pere(inst7);
Partitionneur_Sous_Domaine inst8;verifie_pere(inst8);
Partitionneur_Sous_Zones inst9;verifie_pere(inst9);
Partitionneur_Tranche inst10;verifie_pere(inst10);
Partitionneur_Union inst11;verifie_pere(inst11);
}
