import React from 'react'
import Card from '@mui/material/Card';
import CardActions from '@mui/material/CardActions';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Brightness1RoundedIcon from '@mui/icons-material/Brightness1Rounded';
import Button from '@mui/material/Button';

const Result = ({hidden, scoreResult}) => {

    function Score({ scoreResult }) {
        switch (scoreResult) {
            case null:
            return <></>
            default:
            return <BasicCard score={scoreResult} />
        }
    }

    function BasicCard({ score }) {
        let val
        if (score >= 2) {
          val = <Typography variant="h6" component="div" color="#079615">
            Score de cohérence : {score}
          </Typography>
        }
        else if (score >= 1) {
          val = <Typography variant="h6" component="div" color="#FF9A02">
            Score de cohérence : {score}
          </Typography>
        }
        else if (score >= 0) {
          val = <Typography variant="h6" component="div" color="#E33A3A">
            Score de cohérence : {score}
          </Typography>
        }
        else {
          val = <Typography variant="h6" component="div">
            Score de coherence : {score}
          </Typography>
        }
        return (
          <div className='result'>
            <Card sx={{ minWidth: 275, border: 1 }}>
              <CardContent>
                <Typography variant="h6" component="div">
                  {val}
                </Typography>
              </CardContent>
              <CardActions sx={{ position: 'relative' }}>
                <Button size="small">Voir plus</Button>
              </CardActions>
            </Card>
          </div>
        );
    }

  return (
    <div id='evalSection' hidden={hidden}>
        <Score scoreResult={scoreResult} />
        <div className='cards-key'>
        <div className='card1'>
            <div><Brightness1RoundedIcon sx={{ color: "#079615" }} /></div>
            <p id='scoreCard'>2 - 3 (élevé)</p>
        </div>
        <div className='card1'>
            <div><Brightness1RoundedIcon sx={{ color: "#FF9A02" }} /></div>
            <p id='scoreCard'>1 - 2 (moyen)</p>
        </div>
        <div className='card1'>
            <div><Brightness1RoundedIcon sx={{ color: "#E33A3A" }} /></div>
            <p id='scoreCard'>0 - 1 (bas)</p>
        </div>
        </div>
    </div>
  )
}

export default Result