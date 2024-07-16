import React from 'react'
import './footer.css'
// import footer_logo from '../images/Assets/logo_big.png'
import footer_logo from "../images/Assets/online-shop.jpg";

import instagram_icon from '../images/Assets/instagram_icon.png'
import pintester_icon from '../images/Assets/pintester_icon.png'
import whatsapp_icon from '../images/Assets/whatsapp_icon.png'

export default function Footer() {
  return (
    <div className='footer'>
      <div className='footer-logo'>
        <img src={footer_logo} alt="" />
        <p>SHOPIFY</p>
      </div>
<div>
  
      <ul className='footer-links'>
        <li>Company</li>
        <li>Who are we</li>
        
        <li>Join our team</li>
        <li>Terms and Conditions</li>
      </ul>
     
      <ul className='footer-links'>
        <li>Products</li>
        <li>Mens</li>
        <li>Women</li>
        <li>Kids</li>
      </ul>
      <ul className='footer-links'>
        <li>About</li>
        
        <li>Setting</li>
        <li>Report Abuse</li>
        <li>Help</li>
      </ul>
      <ul className='footer-links'>
        <li>Contact</li>
        <li> New York, NY 10012, US</li>
        <li>shopifyonlineclothorder@gamil.com</li>
        <li>+ 01 234 567 88</li>
        
        
        
       
      </ul>
      </div>
      <div className="footer-social-icon">
        <div className="footer-icons-container">
        <img src={instagram_icon} alt=""  /></div>
        <div className="footer-icons-container">
        <img src={pintester_icon} alt=""  /></div>
        <div className="footer-icons-container">
        <img src={whatsapp_icon} alt=""  /></div>
        </div>
    <div className="footer-copyright">
      <hr />
      <p>Copyright @ 2023 -All Right Reserved</p>
    </div>
    </div>

  )
}
