import React from 'react'
import Typography from '@mui/material/Typography';

const DescriptionSemRel = () => {
  return (
    <div>
      <Typography variant="body2" sx={{ fontFamily: 'Poppins', fontWeight: 300 }}>
          <b>SEMREL</b> est un modèle neuronal à base de <b>LSTM</b> (Long Short Term Memory)
          qui évalue la cohérence d'un discours à deux niveaux sémantiques combinés :
          Phrases <b>(SENTAVG)</b> et Paragraphes <b>(PARSEQ)</b>.
      </Typography>
      <br/>
      <Typography variant="body2" sx={{ fontFamily: 'Poppins', fontWeight: 700, color: '#079615' }}>
          Niveau d'analyse : Sémantique
      </Typography>
    </div>
  )
}

export default DescriptionSemRel