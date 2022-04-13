import React from 'react'
import Typography from '@mui/material/Typography';

const DescriptionParSeq = () => {
  return (
    <div>
      <Typography variant="body2" sx={{ fontFamily: 'Poppins', fontWeight: 300 }}>
          <b>PARSEQ</b> est un modèle neuronal à base de <b>LSTM</b> (Long Short Term Memory)
              qui évalue la cohérence d'un discours à travers les similarités
              cosines entre <b>ses paragraphes</b> (i.e à un niveau plus global),
              en modélisant ainsi la parenté sémantique entre elles
      </Typography>
      <br/>
      <Typography variant="body2" sx={{ fontFamily: 'Poppins', fontWeight: 700, color: '#079615' }}>
          Niveau d'analyse : Sémantique
      </Typography>
    </div>
    )
}

export default DescriptionParSeq