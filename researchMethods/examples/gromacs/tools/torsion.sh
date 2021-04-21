#!/bin/bash

WORKDIR=$PWD
DESTDIR=${WORKDIR}/ref_$1
TOP=${WORKDIR}/top/biphenyl_reference.top
MDP=${WORKDIR}/mdp/em.mdp
GRODIR=${WORKDIR}/gro
BASENAME="biphenyl"
DESTNAME=${DESTDIR}/${BASENAME}

rm -r ${DESTDIR}
mkdir -p ${DESTDIR}

for ANG in {-090..0085..05}
do
	sed -e "/#define ANG/s/\$/ ${ANG}/" ${TOP} > ${DESTNAME}_${ANG}.top
	gmx grompp -f ${MDP} -p ${DESTNAME}_${ANG}.top -c ${GRODIR}/${BASENAME}_${ANG}.gro -o ${DESTNAME}_${ANG} &&
	gmx mdrun -deffnm ${DESTNAME}_${ANG}
done
